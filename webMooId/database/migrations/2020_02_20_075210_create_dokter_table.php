<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreateDokterTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('dokter', function (Blueprint $table) {
            $table->bigIncrements('id');
            $table->unsignedBigInteger('id_peternakan');
            $table->string('nama');
            $table->string('email');
            $table->string('no_telepon');
            $table->string('pendidikan_terakhir');
            $table->integer('pengalaman_kerja');
            $table->text('tentang')->nullable();
            $table->string('avatar')->nullable();
            $table->string('tempat_kerja');
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
        Schema::dropIfExists('dokter');
    }
}
