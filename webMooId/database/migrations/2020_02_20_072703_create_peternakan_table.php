<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreatePeternakanTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('peternakan', function (Blueprint $table) {
            $table->bigIncrements('id');
            $table->string('nama_peternakan');
            $table->string('username');
            $table->string('password');
            $table->enum('jenis_peternakan', ['perorang', 'kelompok', 'perusahaan']);
            $table->string('nama_lengkap');
            $table->string('no_telepon');
            $table->string('email');
            $table->string('alamat_lengkap');
            $table->string('latitude');
            $table->string('longitude');
            $table->string('area')->nullable();
            $table->string('avatar')->nullable();
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('peternakan');
    }
}
