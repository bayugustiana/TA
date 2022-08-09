<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreateRekamMedisTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('rekam_medis', function (Blueprint $table) {
            $table->bigIncrements('id');
            $table->unsignedBigInteger('id_obat');
            $table->unsignedBigInteger('id_dokter');
            $table->unsignedBigInteger('id_sapi');
            $table->unsignedBigInteger('id_peternakan');
            $table->string('nama_pesan');
            $table->dateTime('waktu');
            $table->foreign('id_obat')->references('id')->on('obat')->onUpdate('cascade')->onDelete('cascade');
            $table->foreign('id_dokter')->references('id')->on('dokter')->onUpdate('cascade')->onDelete('cascade');
            $table->foreign('id_sapi')->references('id')->on('sapi')->onUpdate('cascade')->onDelete('cascade');
            $table->foreign('id_peternakan')->references('id')->on('peternakan')->onUpdate('cascade')->onDelete('cascade');
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('rekam_medis');
    }
}
