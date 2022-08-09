<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreateSapiTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('sapi', function (Blueprint $table) {
            $table->bigIncrements('id');
            $table->unsignedBigInteger('id_tempat');
            $table->date('tanggal_lahir');
            $table->string('jenis_sapi');
            $table->string('tipe_sapi');
            $table->enum('jenis_kelamin', ['laki-laki','perempuan']);
            $table->foreign('id_tempat')->references('id')->on('tempat')->onUpdate('cascade')->onDelete('cascade');
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('sapi');
    }
}
